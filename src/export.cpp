#include "export.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cluster.h"
#include "json/json.h"
#include "layer.h"
#include "network.h"
#include "noc.h"
#include "schnode.h"

namespace {

struct Box {
	bool valid = false;
	std::array<len_t, 4> lower{};
	std::array<len_t, 4> upper{};
};

struct WorkloadMeta {
	SchNode::wlid_t workload_id = 0;
	int core_id = -1;
	int chiplet_id = -1;
	cycle_t start_time = 0;
	cycle_t duration = 0;
	cycle_t end_time = 0;
	const Json::Value* workload = nullptr;
};

struct ComputeAggregate {
	int chiplet_id = -1;
	std::string layer_name;
	std::string layer_type;
	cycle_t start_time = 0;
	cycle_t duration = 0;
	Box ofmap_box;
	Box ifmap_box;
	vol_t exact_ofmap_elements = 0;
	std::vector<const WorkloadMeta*> members;
};

struct TransferAggregate {
	int chiplet_id = -1;
	cycle_t time = 0;
	int type_rank = 0;
	std::string event_type;
	std::string tensor_type;
	std::string layer_name;
	std::string peer_kind;
	int peer_chiplet_id = -1;
	std::uint64_t bytes = 0;
	std::set<unsigned int> transfer_ids;
	std::set<unsigned int> workload_ids;
};

struct GenericEvent {
	cycle_t sort_time = 0;
	int type_rank = 0;
	Json::Value payload;
};

struct ScalesimRow {
	int chiplet_id = -1;
	cycle_t start_time = 0;
	std::string row_name;
	len_t ifmap_h = 0;
	len_t ifmap_w = 0;
	len_t filt_h = 0;
	len_t filt_w = 0;
	len_t channels = 0;
	len_t num_filters = 0;
	len_t stride_h = 0;
	len_t stride_w = 0;
	len_t batch_size = 0;
};

enum class WorkloadPhase {
	Read,
	Compute,
	Write,
	Done
};

struct EventRuntime {
	int event_id = -1;
	int chiplet_id = -1;
	cycle_t sort_time = 0;
	int type_rank = 0;
	Json::Value payload;
	std::vector<SchNode::wlid_t> workload_ids;
};

struct WorkloadRuntime {
	SchNode::wlid_t workload_id = 0;
	int core_id = -1;
	std::set<int> pending_read_events;
	int compute_event_id = -1;
	bool compute_done = false;
	std::set<int> pending_write_events;
};

struct CoreRuntime {
	int core_id = -1;
	int chiplet_id = -1;
	std::vector<SchNode::wlid_t> workload_order;
	size_t current_index = 0;
};

void sort_chiplet_event_vectors(std::map<int, std::vector<GenericEvent>>& chiplet_events) {
	for(auto& [chiplet_id, events] : chiplet_events) {
		(void)chiplet_id;
		std::stable_sort(events.begin(), events.end(), [](const GenericEvent& lhs, const GenericEvent& rhs) {
			return std::tie(lhs.sort_time, lhs.type_rank) < std::tie(rhs.sort_time, rhs.type_rank);
		});
	}
}

std::vector<SchNode::wlid_t> event_workload_ids(const Json::Value& payload) {
	std::vector<SchNode::wlid_t> workload_ids;
	const Json::Value& json_ids = payload["workload_ids"];
	if(!json_ids.isArray()) {
		return workload_ids;
	}
	workload_ids.reserve(json_ids.size());
	for(Json::Value::ArrayIndex i = 0; i < json_ids.size(); ++i) {
		workload_ids.push_back(json_ids[i].asUInt());
	}
	return workload_ids;
}

std::vector<EventRuntime> flatten_chiplet_events(const std::map<int, std::vector<GenericEvent>>& chiplet_events) {
	std::vector<EventRuntime> flattened;
	int event_id = 0;
	for(const auto& [chiplet_id, events] : chiplet_events) {
		for(const GenericEvent& event : events) {
			EventRuntime runtime;
			runtime.event_id = event_id++;
			runtime.chiplet_id = chiplet_id;
			runtime.sort_time = event.sort_time;
			runtime.type_rank = event.type_rank;
			runtime.payload = event.payload;
			runtime.workload_ids = event_workload_ids(event.payload);
			flattened.push_back(std::move(runtime));
		}
	}
	return flattened;
}

WorkloadPhase workload_phase(const WorkloadRuntime& workload) {
	if(!workload.pending_read_events.empty()) {
		return WorkloadPhase::Read;
	}
	if(!workload.compute_done) {
		return WorkloadPhase::Compute;
	}
	if(!workload.pending_write_events.empty()) {
		return WorkloadPhase::Write;
	}
	return WorkloadPhase::Done;
}

void advance_core(CoreRuntime& core, const std::map<SchNode::wlid_t, WorkloadRuntime>& workloads) {
	while(core.current_index < core.workload_order.size()) {
		const auto it = workloads.find(core.workload_order[core.current_index]);
		if(it == workloads.end()) {
			throw std::runtime_error("core runtime references unknown workload");
		}
		if(workload_phase(it->second) != WorkloadPhase::Done) {
			return;
		}
		++core.current_index;
	}
}

std::vector<int> current_candidate_event_ids(
	CoreRuntime& core,
	const std::map<SchNode::wlid_t, WorkloadRuntime>& workloads
) {
	advance_core(core, workloads);
	if(core.current_index >= core.workload_order.size()) {
		return {};
	}

	const auto it = workloads.find(core.workload_order[core.current_index]);
	if(it == workloads.end()) {
		throw std::runtime_error("current core workload state is missing");
	}
	const WorkloadRuntime& workload = it->second;
	switch(workload_phase(workload)) {
	case WorkloadPhase::Read:
		return {workload.pending_read_events.begin(), workload.pending_read_events.end()};
	case WorkloadPhase::Compute:
		return workload.compute_event_id >= 0 ? std::vector<int>{workload.compute_event_id} : std::vector<int>{};
	case WorkloadPhase::Write:
		return {workload.pending_write_events.begin(), workload.pending_write_events.end()};
	case WorkloadPhase::Done:
		break;
	}
	return {};
}

bool transfer_ids_equal(const Json::Value& lhs, const Json::Value& rhs) {
	if(!lhs.isArray() || !rhs.isArray() || lhs.size() != rhs.size()) {
		return false;
	}
	for(Json::Value::ArrayIndex i = 0; i < lhs.size(); ++i) {
		if(lhs[i].asUInt() != rhs[i].asUInt()) {
			return false;
		}
	}
	return true;
}

bool event_is_directly_poppable(const EventRuntime& event) {
	const std::string type = event.payload["type"].asString();
	if(type == "compute") {
		return true;
	}
	return event.payload["peer"]["kind"].asString() == "dram";
}

bool event_pair_matches(const EventRuntime& lhs, const EventRuntime& rhs) {
	const std::string lhs_type = lhs.payload["type"].asString();
	const std::string rhs_type = rhs.payload["type"].asString();
	if(!((lhs_type == "read" && rhs_type == "write") || (lhs_type == "write" && rhs_type == "read"))) {
		return false;
	}
	if(lhs.payload["peer"]["kind"].asString() != "chiplet" || rhs.payload["peer"]["kind"].asString() != "chiplet") {
		return false;
	}
	return lhs.payload["peer"]["chiplet_id"].asInt() == rhs.chiplet_id &&
		rhs.payload["peer"]["chiplet_id"].asInt() == lhs.chiplet_id &&
		transfer_ids_equal(lhs.payload["transfer_ids"], rhs.payload["transfer_ids"]);
}

bool event_ready(
	const EventRuntime& event,
	std::map<SchNode::wlid_t, WorkloadRuntime>& workloads,
	std::map<int, CoreRuntime>& cores
) {
	if(event.workload_ids.empty()) {
		return false;
	}

	std::set<int> touched_cores;
	for(SchNode::wlid_t workload_id : event.workload_ids) {
		const auto workload_it = workloads.find(workload_id);
		if(workload_it == workloads.end()) {
			throw std::runtime_error("event references unknown workload id");
		}
		touched_cores.insert(workload_it->second.core_id);
	}
	for(int core_id : touched_cores) {
		auto core_it = cores.find(core_id);
		if(core_it == cores.end()) {
			throw std::runtime_error("event references unknown core runtime");
		}
		advance_core(core_it->second, workloads);
	}

	const std::string type = event.payload["type"].asString();
	for(SchNode::wlid_t workload_id : event.workload_ids) {
		const WorkloadRuntime& workload = workloads.at(workload_id);
		const CoreRuntime& core = cores.at(workload.core_id);
		if(core.current_index >= core.workload_order.size() || core.workload_order[core.current_index] != workload_id) {
			return false;
		}
		switch(workload_phase(workload)) {
		case WorkloadPhase::Read:
			if(type != "read" || workload.pending_read_events.count(event.event_id) == 0) {
				return false;
			}
			break;
		case WorkloadPhase::Compute:
			if(type != "compute" || workload.compute_done || workload.compute_event_id != event.event_id) {
				return false;
			}
			break;
		case WorkloadPhase::Write:
			if(type != "write" || workload.pending_write_events.count(event.event_id) == 0) {
				return false;
			}
			break;
		case WorkloadPhase::Done:
			return false;
		}
	}
	return true;
}

void pop_event(
	const EventRuntime& event,
	std::map<SchNode::wlid_t, WorkloadRuntime>& workloads,
	std::map<int, CoreRuntime>& cores
) {
	const std::string type = event.payload["type"].asString();
	std::set<int> touched_cores;
	for(SchNode::wlid_t workload_id : event.workload_ids) {
		auto workload_it = workloads.find(workload_id);
		if(workload_it == workloads.end()) {
			throw std::runtime_error("cannot pop event for unknown workload");
		}
		WorkloadRuntime& workload = workload_it->second;
		touched_cores.insert(workload.core_id);
		if(type == "read") {
			workload.pending_read_events.erase(event.event_id);
		} else if(type == "compute") {
			if(workload.compute_event_id != event.event_id) {
				throw std::runtime_error("compute event does not match workload runtime");
			}
			workload.compute_done = true;
		} else if(type == "write") {
			workload.pending_write_events.erase(event.event_id);
		} else {
			throw std::runtime_error("unknown event type while popping event");
		}
	}
	for(int core_id : touched_cores) {
		advance_core(cores.at(core_id), workloads);
	}
}

std::map<int, std::vector<int>> build_runtime_states(
	const std::map<SchNode::wlid_t, WorkloadMeta>& workload_meta,
	const std::vector<EventRuntime>& events,
	std::map<SchNode::wlid_t, WorkloadRuntime>& workloads,
	std::map<int, CoreRuntime>& cores
) {
	std::map<int, std::vector<std::pair<cycle_t, SchNode::wlid_t>>> workloads_by_core;
	for(const auto& [workload_id, meta] : workload_meta) {
		WorkloadRuntime state;
		state.workload_id = workload_id;
		state.core_id = meta.core_id;
		workloads.emplace(workload_id, std::move(state));
		workloads_by_core[meta.core_id].push_back({meta.start_time, workload_id});
	}

	for(const EventRuntime& event : events) {
		const std::string type = event.payload["type"].asString();
		for(SchNode::wlid_t workload_id : event.workload_ids) {
			auto workload_it = workloads.find(workload_id);
			if(workload_it == workloads.end()) {
				throw std::runtime_error("event payload references an unknown workload id");
			}
			WorkloadRuntime& workload = workload_it->second;
			if(type == "read") {
				workload.pending_read_events.insert(event.event_id);
			} else if(type == "compute") {
				if(workload.compute_event_id >= 0) {
					throw std::runtime_error("workload is associated with multiple compute events");
				}
				workload.compute_event_id = event.event_id;
			} else if(type == "write") {
				workload.pending_write_events.insert(event.event_id);
			}
		}
	}

	std::map<int, std::vector<int>> chiplet_cores;
	for(auto& [core_id, core_workloads] : workloads_by_core) {
		std::sort(core_workloads.begin(), core_workloads.end(), [](const auto& lhs, const auto& rhs) {
			return std::tie(lhs.first, lhs.second) < std::tie(rhs.first, rhs.second);
		});
		CoreRuntime core;
		core.core_id = core_id;
		for(const auto& [start_time, workload_id] : core_workloads) {
			(void)start_time;
			core.workload_order.push_back(workload_id);
		}
		if(core.workload_order.empty()) {
			continue;
		}
		const auto meta_it = workload_meta.find(core.workload_order.front());
		if(meta_it == workload_meta.end()) {
			throw std::runtime_error("missing workload metadata while building core runtime");
		}
		core.chiplet_id = meta_it->second.chiplet_id;
		cores.emplace(core_id, core);
		chiplet_cores[core.chiplet_id].push_back(core_id);
	}

	for(auto& [chiplet_id, core_ids] : chiplet_cores) {
		(void)chiplet_id;
		std::sort(core_ids.begin(), core_ids.end());
	}
	for(const auto& [workload_id, workload] : workloads) {
		(void)workload_id;
		if(workload.compute_event_id < 0) {
			throw std::runtime_error("workload runtime is missing a compute event");
		}
	}
	return chiplet_cores;
}

std::vector<int> chiplet_candidate_event_ids(
	int chiplet_id,
	const std::map<int, std::vector<int>>& chiplet_cores,
	std::map<int, CoreRuntime>& cores,
	const std::map<SchNode::wlid_t, WorkloadRuntime>& workloads
) {
	std::set<int> candidate_ids;
	const auto chiplet_it = chiplet_cores.find(chiplet_id);
	if(chiplet_it == chiplet_cores.end()) {
		return {};
	}
	for(int core_id : chiplet_it->second) {
		auto core_it = cores.find(core_id);
		if(core_it == cores.end()) {
			throw std::runtime_error("chiplet runtime references unknown core");
		}
		const auto candidates = current_candidate_event_ids(core_it->second, workloads);
		candidate_ids.insert(candidates.begin(), candidates.end());
	}
	return {candidate_ids.begin(), candidate_ids.end()};
}

std::map<int, std::vector<GenericEvent>> build_pairable_chiplet_events(
	const std::map<int, std::vector<GenericEvent>>& raw_chiplet_events,
	const std::map<SchNode::wlid_t, WorkloadMeta>& workload_meta
) {
	auto sorted_events_by_chiplet = raw_chiplet_events;
	sort_chiplet_event_vectors(sorted_events_by_chiplet);
	const auto events = flatten_chiplet_events(sorted_events_by_chiplet);

	std::map<SchNode::wlid_t, WorkloadRuntime> workloads;
	std::map<int, CoreRuntime> cores;
	const auto chiplet_cores = build_runtime_states(workload_meta, events, workloads, cores);

	std::map<int, std::vector<GenericEvent>> reordered_events;
	size_t popped_event_count = 0;
	while(popped_event_count < events.size()) {
		bool progress = false;

		while(true) {
			bool direct_progress = false;
			for(const auto& [chiplet_id, core_ids] : chiplet_cores) {
				(void)core_ids;
				for(int event_id : chiplet_candidate_event_ids(chiplet_id, chiplet_cores, cores, workloads)) {
					const EventRuntime& event = events.at(static_cast<size_t>(event_id));
					if(!event_is_directly_poppable(event) || !event_ready(event, workloads, cores)) {
						continue;
					}
					pop_event(event, workloads, cores);
					reordered_events[chiplet_id].push_back({event.sort_time, event.type_rank, event.payload});
					++popped_event_count;
					direct_progress = true;
					progress = true;
					break;
				}
			}
			if(!direct_progress) {
				break;
			}
		}

		bool pair_progress = false;
		std::vector<std::pair<int, int>> ready_transfers;
		for(const auto& [chiplet_id, core_ids] : chiplet_cores) {
			(void)core_ids;
			for(int event_id : chiplet_candidate_event_ids(chiplet_id, chiplet_cores, cores, workloads)) {
				const EventRuntime& event = events.at(static_cast<size_t>(event_id));
				if(event_is_directly_poppable(event) || !event_ready(event, workloads, cores)) {
					continue;
				}
				ready_transfers.push_back({chiplet_id, event_id});
			}
		}

		for(size_t i = 0; i < ready_transfers.size() && !pair_progress; ++i) {
			const EventRuntime& lhs = events.at(static_cast<size_t>(ready_transfers[i].second));
			for(size_t j = i + 1; j < ready_transfers.size(); ++j) {
				const EventRuntime& rhs = events.at(static_cast<size_t>(ready_transfers[j].second));
				if(!event_pair_matches(lhs, rhs)) {
					continue;
				}
				pop_event(lhs, workloads, cores);
				pop_event(rhs, workloads, cores);
				reordered_events[lhs.chiplet_id].push_back({lhs.sort_time, lhs.type_rank, lhs.payload});
				reordered_events[rhs.chiplet_id].push_back({rhs.sort_time, rhs.type_rank, rhs.payload});
				popped_event_count += 2;
				pair_progress = true;
				progress = true;
				break;
			}
		}

		if(!progress) {
			std::ostringstream oss;
			oss << "failed to reconstruct a fully pairable chiplet event order; "
				<< "popped " << popped_event_count << " of " << events.size() << " events";
			throw std::runtime_error(oss.str());
		}
	}

	return reordered_events;
}

Json::Value json_u64(std::uint64_t value) {
	return Json::Value(static_cast<double>(value));
}

std::string sanitize_name(const std::string& value) {
	std::string out = value;
	for(char& ch : out) {
		if(!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-')) {
			ch = '_';
		}
	}
	return out;
}

std::uint64_t bits_to_bytes(const Json::Value& value) {
	const double bits = value.asDouble();
	return static_cast<std::uint64_t>((bits + 7.0) / 8.0);
}

std::string u64_to_string(std::uint64_t value) {
	std::ostringstream oss;
	oss << value;
	return oss.str();
}

std::filesystem::path output_dir_from_hint(const std::string& output_hint_path) {
	if(output_hint_path.empty()) {
		return std::filesystem::path("output") / "exports";
	}
	std::filesystem::path hint(output_hint_path);
	std::filesystem::path parent = hint.parent_path();
	if(parent.empty()) {
		parent = ".";
	}
	return parent / "exports";
}

bool is_digits_only(const std::string& text) {
	if(text.empty()) {
		return false;
	}
	for(unsigned char ch : text) {
		if(std::isdigit(ch) == 0) {
			return false;
		}
	}
	return true;
}

bool is_core_key(const std::string& key) {
	return key != "-1" && is_digits_only(key);
}

std::vector<std::string> get_sorted_core_keys(const Json::Value& ir) {
	std::vector<std::string> keys;
	for(const std::string& key : ir.getMemberNames()) {
		if(is_core_key(key)) {
			keys.push_back(key);
		}
	}
	std::sort(keys.begin(), keys.end(), [](const std::string& lhs, const std::string& rhs) {
		return std::stoi(lhs) < std::stoi(rhs);
	});
	return keys;
}

Box box_from_lower_upper(const Json::Value& lower, const Json::Value& upper) {
	Box box;
	if(!lower.isArray() || !upper.isArray() || lower.size() != 4 || upper.size() != 4) {
		return box;
	}
	box.valid = true;
	for(Json::Value::ArrayIndex i = 0; i < 4; ++i) {
		box.lower[i] = lower[i].asUInt();
		box.upper[i] = upper[i].asUInt();
	}
	return box;
}

Box workload_ofmap_box(const Json::Value& workload) {
	return box_from_lower_upper(workload["workload"][0u], workload["workload"][1u]);
}

Box workload_ifmap_box(const Json::Value& workload) {
	if(workload.isMember("ifmap")) {
		return box_from_lower_upper(workload["ifmap"]["lower"], workload["ifmap"]["upper"]);
	}
	return Box{};
}

void merge_box(Box& dst, const Box& src) {
	if(!src.valid) {
		return;
	}
	if(!dst.valid) {
		dst = src;
		return;
	}
	for(size_t i = 0; i < dst.lower.size(); ++i) {
		dst.lower[i] = std::min(dst.lower[i], src.lower[i]);
		dst.upper[i] = std::max(dst.upper[i], src.upper[i]);
	}
}

vol_t box_elements(const Box& box) {
	if(!box.valid) {
		return 0;
	}
	vol_t total = 1;
	for(size_t i = 0; i < box.lower.size(); ++i) {
		total *= static_cast<vol_t>(box.upper[i] - box.lower[i] + 1);
	}
	return total;
}

Json::Value box_to_json(const Box& box) {
	Json::Value value;
	if(!box.valid) {
		return value;
	}
	for(size_t i = 0; i < box.lower.size(); ++i) {
		value["lower"].append(box.lower[i]);
		value["upper"].append(box.upper[i]);
	}
	return value;
}

struct ChipletCoord {
	int chiplet_id = -1;
	int chiplet_x = -1;
	int chiplet_y = -1;
};

ChipletCoord core_to_chiplet(int core_id) {
	const int span = static_cast<int>(Cluster::xlen) + 2;
	const int raw = core_id - 1;
	const int x = raw % span;
	const int y = raw / span;
	const int chiplet_x = x / static_cast<int>(NoC::x_step);
	const int chiplet_y = y / static_cast<int>(NoC::y_step);
	return {
		chiplet_y * static_cast<int>(NoC::x_cut) + chiplet_x,
		chiplet_x,
		chiplet_y
	};
}

std::map<std::string, Network::lid_t> build_layer_index(const Network& network_ref) {
	std::map<std::string, Network::lid_t> index;
	for(Network::lid_t i = 0; i < network_ref.len(); ++i) {
		index[network_ref.getNode(i).name()] = i;
	}
	return index;
}

Json::Value layer_shape_json(const Layer& layer) {
	Json::Value shape;
	if(REF_IS_INSTANCE(layer, GroupConvLayer)) {
		const auto& wl = static_cast<const GroupConvLayer&>(layer).get_workload();
		shape["kind"] = "group_conv2d";
		shape["groups"] = wl.G;
		shape["channels"] = wl.C;
		shape["num_filters"] = wl.K;
		shape["ofmap_h"] = wl.H;
		shape["ofmap_w"] = wl.W;
		shape["filter_h"] = wl.R;
		shape["filter_w"] = wl.S;
		shape["stride_h"] = wl.sH;
		shape["stride_w"] = wl.sW;
		return shape;
	}
	if(REF_IS_INSTANCE(layer, FCLayer)) {
		const auto& wl = static_cast<const ConvLayer&>(layer).get_workload();
		shape["kind"] = "fc";
		shape["channels"] = wl.C;
		shape["num_filters"] = wl.K;
		shape["ifmap_h"] = wl.R;
		shape["ifmap_w"] = wl.S;
		return shape;
	}
	if(REF_IS_INSTANCE(layer, ConvLayer)) {
		const auto& wl = static_cast<const ConvLayer&>(layer).get_workload();
		shape["kind"] = "conv2d";
		shape["channels"] = wl.C;
		shape["num_filters"] = wl.K;
		shape["ofmap_h"] = wl.H;
		shape["ofmap_w"] = wl.W;
		shape["filter_h"] = wl.R;
		shape["filter_w"] = wl.S;
		shape["stride_h"] = wl.sH;
		shape["stride_w"] = wl.sW;
		return shape;
	}
	shape["kind"] = "other";
	shape["ifmap_size"] = json_u64(layer.real_ifmap_shape().size);
	shape["ofmap_channels"] = layer.ofmap_shape().c;
	shape["ofmap_h"] = layer.ofmap_shape().h;
	shape["ofmap_w"] = layer.ofmap_shape().w;
	return shape;
}

std::vector<ScalesimRow> csv_rows_from_aggregate(
	const ComputeAggregate& aggregate,
	const Layer& layer
) {
	std::vector<ScalesimRow> rows;
	if(!aggregate.ofmap_box.valid || !aggregate.ifmap_box.valid) {
		return rows;
	}

	const std::string row_name =
		"chiplet" + std::to_string(aggregate.chiplet_id) + "_" +
		sanitize_name(aggregate.layer_name) + "_t" + std::to_string(aggregate.start_time);

	if(REF_IS_INSTANCE(layer, GroupConvLayer)) {
		const auto& wl = static_cast<const GroupConvLayer&>(layer).get_workload();
		const len_t ofmap_c_from = aggregate.ofmap_box.lower[1];
		const len_t ofmap_c_to = aggregate.ofmap_box.upper[1] + 1;
		const len_t first_group = ofmap_c_from / wl.GK;
		const len_t last_group = (ofmap_c_to - 1) / wl.GK;
		for(len_t group_id = first_group; group_id <= last_group; ++group_id) {
			const len_t group_lo = std::max(ofmap_c_from, group_id * wl.GK);
			const len_t group_hi = std::min(ofmap_c_to, (group_id + 1) * wl.GK);
			if(group_lo >= group_hi) {
				continue;
			}
			ScalesimRow row;
			row.chiplet_id = aggregate.chiplet_id;
			row.start_time = aggregate.start_time;
			row.row_name = row_name + "_g" + std::to_string(group_id);
			row.ifmap_h = aggregate.ifmap_box.upper[2] - aggregate.ifmap_box.lower[2] + 1;
			row.ifmap_w = aggregate.ifmap_box.upper[3] - aggregate.ifmap_box.lower[3] + 1;
			row.filt_h = wl.R;
			row.filt_w = wl.S;
			row.channels = wl.GC;
			row.num_filters = group_hi - group_lo;
			row.stride_h = wl.sH;
			row.stride_w = wl.sW;
			row.batch_size = aggregate.ofmap_box.upper[0] - aggregate.ofmap_box.lower[0] + 1;
			rows.push_back(row);
		}
		return rows;
	}

	if(REF_IS_INSTANCE(layer, ConvLayer)) {
		const auto& wl = static_cast<const ConvLayer&>(layer).get_workload();
		ScalesimRow row;
		row.chiplet_id = aggregate.chiplet_id;
		row.start_time = aggregate.start_time;
		row.row_name = row_name;
		row.ifmap_h = aggregate.ifmap_box.upper[2] - aggregate.ifmap_box.lower[2] + 1;
		row.ifmap_w = aggregate.ifmap_box.upper[3] - aggregate.ifmap_box.lower[3] + 1;
		row.filt_h = wl.R;
		row.filt_w = wl.S;
		row.channels = aggregate.ifmap_box.upper[1] - aggregate.ifmap_box.lower[1] + 1;
		row.num_filters = aggregate.ofmap_box.upper[1] - aggregate.ofmap_box.lower[1] + 1;
		row.stride_h = wl.sH;
		row.stride_w = wl.sW;
		row.batch_size = aggregate.ofmap_box.upper[0] - aggregate.ofmap_box.lower[0] + 1;
		rows.push_back(row);
	}
	return rows;
}

void write_scalesim_csv(const std::filesystem::path& csv_path, const std::vector<ScalesimRow>& rows) {
	std::ofstream out(csv_path);
	out << "Layer name,IFMAP Height,IFMAP Width,Filter Height,Filter Width,Channels,Num Filter,Stride Height,Stride Width,Batch Size,\n";
	for(const ScalesimRow& row : rows) {
		out << row.row_name << ','
			<< row.ifmap_h << ','
			<< row.ifmap_w << ','
			<< row.filt_h << ','
			<< row.filt_w << ','
			<< row.channels << ','
			<< row.num_filters << ','
			<< row.stride_h << ','
			<< row.stride_w << ','
			<< row.batch_size << ",\n";
	}
}

std::string join_json_array(const Json::Value& array) {
	std::ostringstream oss;
	for(Json::Value::ArrayIndex i = 0; i < array.size(); ++i) {
		if(i != 0) {
			oss << ", ";
		}
		oss << array[i].asUInt();
	}
	return oss.str();
}

std::string format_box_text(const Json::Value& box) {
	if(!box.isObject() || !box.isMember("lower") || !box.isMember("upper")) {
		return "-";
	}
	std::ostringstream oss;
	oss << '[';
	for(Json::Value::ArrayIndex i = 0; i < box["lower"].size(); ++i) {
		if(i != 0) {
			oss << ", ";
		}
		oss << box["lower"][i].asUInt() << ':' << box["upper"][i].asUInt();
	}
	oss << ']';
	return oss.str();
}

std::string format_layer_shape_text(const Json::Value& shape) {
	if(!shape.isObject() || !shape.isMember("kind")) {
		return "-";
	}
	std::ostringstream oss;
	const std::string kind = shape["kind"].asString();
	oss << kind;
	if(kind == "group_conv2d") {
		oss << "(G=" << shape["groups"].asUInt()
			<< ", C=" << shape["channels"].asUInt()
			<< ", K=" << shape["num_filters"].asUInt()
			<< ", H=" << shape["ofmap_h"].asUInt()
			<< ", W=" << shape["ofmap_w"].asUInt()
			<< ", R=" << shape["filter_h"].asUInt()
			<< ", S=" << shape["filter_w"].asUInt()
			<< ", sH=" << shape["stride_h"].asUInt()
			<< ", sW=" << shape["stride_w"].asUInt()
			<< ')';
		return oss.str();
	}
	if(kind == "conv2d") {
		oss << "(C=" << shape["channels"].asUInt()
			<< ", K=" << shape["num_filters"].asUInt()
			<< ", H=" << shape["ofmap_h"].asUInt()
			<< ", W=" << shape["ofmap_w"].asUInt()
			<< ", R=" << shape["filter_h"].asUInt()
			<< ", S=" << shape["filter_w"].asUInt()
			<< ", sH=" << shape["stride_h"].asUInt()
			<< ", sW=" << shape["stride_w"].asUInt()
			<< ')';
		return oss.str();
	}
	if(kind == "fc") {
		oss << "(C=" << shape["channels"].asUInt()
			<< ", K=" << shape["num_filters"].asUInt()
			<< ", IFMAP_H=" << shape["ifmap_h"].asUInt()
			<< ", IFMAP_W=" << shape["ifmap_w"].asUInt()
			<< ')';
		return oss.str();
	}
	if(kind == "other") {
		oss << "(ifmap_size=" << u64_to_string(static_cast<std::uint64_t>(shape["ifmap_size"].asDouble()))
			<< ", ofmap_c=" << shape["ofmap_channels"].asUInt()
			<< ", ofmap_h=" << shape["ofmap_h"].asUInt()
			<< ", ofmap_w=" << shape["ofmap_w"].asUInt()
			<< ')';
		return oss.str();
	}
	return kind;
}

void write_chiplet_timeline_text(const std::filesystem::path& txt_path, const Json::Value& chiplet_json) {
	std::ofstream out(txt_path);
	const Json::Value& metadata = chiplet_json["metadata"];
	out << "Gemini Chiplet Timeline\n";
	out << "network_name: " << metadata["network_name"].asString() << '\n';
	out << "mesh: " << metadata["xlen"].asUInt() << 'x' << metadata["ylen"].asUInt() << '\n';
	out << "chiplet_grid: " << metadata["x_cut"].asUInt() << 'x' << metadata["y_cut"].asUInt() << '\n';
	out << "chiplet_step: " << metadata["x_step"].asUInt() << 'x' << metadata["y_step"].asUInt() << '\n';
	out << "chiplet_count: " << metadata["chiplet_count"].asUInt() << '\n';
	out << "source: " << metadata["source"].asString() << "\n\n";

	for(const Json::Value& chiplet : chiplet_json["chiplets"]) {
		out << "================================================================================\n";
		out << "chiplet " << chiplet["chiplet_id"].asUInt()
			<< " (x=" << chiplet["chiplet_x"].asUInt()
			<< ", y=" << chiplet["chiplet_y"].asUInt() << ")\n";
		out << "cores: [" << join_json_array(chiplet["core_ids"]) << "]\n";
		out << "events:\n";

		for(const Json::Value& event : chiplet["events"]) {
			const std::string type = event["type"].asString();
			if(type == "read") {
				out << "  [t=" << u64_to_string(static_cast<std::uint64_t>(event["time"].asDouble())) << "] "
					<< "READ  "
					<< event["tensor_type"].asString() << ' '
					<< u64_to_string(static_cast<std::uint64_t>(event["bytes"].asDouble())) << " B"
					<< "  from ";
				if(event["peer"]["kind"].asString() == "dram") {
					out << "DRAM";
				} else {
					out << "chiplet " << event["peer"]["chiplet_id"].asUInt();
				}
				out << "  for layer " << event["layer_name"].asString() << '\n';
				continue;
			}

			if(type == "write") {
				out << "  [t=" << u64_to_string(static_cast<std::uint64_t>(event["time"].asDouble())) << "] "
					<< "WRITE "
					<< event["tensor_type"].asString() << ' '
					<< u64_to_string(static_cast<std::uint64_t>(event["bytes"].asDouble())) << " B"
					<< "  to ";
				if(event["peer"]["kind"].asString() == "dram") {
					out << "DRAM";
				} else {
					out << "chiplet " << event["peer"]["chiplet_id"].asUInt();
				}
				out << "  from layer " << event["layer_name"].asString() << '\n';
				continue;
			}

			if(type == "compute") {
				out << "  [t=" << u64_to_string(static_cast<std::uint64_t>(event["time_start"].asDouble()))
					<< " -> " << u64_to_string(static_cast<std::uint64_t>(event["time_end"].asDouble()))
					<< ", dur=" << u64_to_string(static_cast<std::uint64_t>(event["duration"].asDouble())) << "] "
					<< "COMPUTE "
					<< event["layer_name"].asString()
					<< "  type=" << event["layer_type"].asString()
					<< "  shape=" << format_layer_shape_text(event["layer_shape"])
					<< "  cores=[" << join_json_array(event["core_ids"]) << "]"
					<< "  partitions=" << event["partition_count"].asUInt()
					<< '\n';
				out << "    ifmap=" << format_box_text(event["ifmap"])
					<< "  ofmap=" << format_box_text(event["ofmap"]) << '\n';
			}
		}
		out << '\n';
	}
}

template <typename... Args>
std::string transfer_key(Args&&... args) {
	std::ostringstream oss;
	((oss << std::forward<Args>(args) << '|'), ...);
	return oss.str();
}

} // namespace

ExportArtifacts export_chiplet_artifacts(
	const SchNode& schedule,
	const std::string& output_hint_path,
	const std::string& network_name
) {
	const Json::Value ir = schedule.IR_gen();
	const auto core_keys = get_sorted_core_keys(ir);
	const auto layer_index = build_layer_index(*network);

	std::map<SchNode::wlid_t, WorkloadMeta> workload_meta;
	std::map<std::tuple<int, std::string, cycle_t, cycle_t>, ComputeAggregate> compute_aggregates;
	std::map<std::string, TransferAggregate> transfer_aggregates;
	std::map<int, std::vector<GenericEvent>> chiplet_events;
	std::map<int, std::vector<int>> chiplet_core_ids;

	for(int y = 0; y < Cluster::ylen; ++y) {
		for(int x = 0; x < Cluster::xlen; ++x) {
			const int core_id = y * (Cluster::xlen + 2) + x + 1;
			const auto chiplet = core_to_chiplet(core_id);
			chiplet_core_ids[chiplet.chiplet_id].push_back(core_id);
		}
	}

	for(const std::string& core_key : core_keys) {
		const int core_id = std::stoi(core_key);
		const int chiplet_id = core_to_chiplet(core_id).chiplet_id;
		cycle_t cursor = 0;
		for(const Json::Value& workload : ir[core_key]) {
			const SchNode::wlid_t workload_id = workload["workload_id"].asUInt();
			const cycle_t duration = workload["time"].asUInt();
			WorkloadMeta meta{
				workload_id,
				core_id,
				chiplet_id,
				cursor,
				duration,
				cursor + duration,
				&workload
			};
			workload_meta.emplace(workload_id, meta);
			cursor += duration;

			auto key = std::make_tuple(
				chiplet_id,
				workload["layer_name"].asString(),
				meta.start_time,
				meta.duration
			);
			auto& aggregate = compute_aggregates[key];
			if(aggregate.members.empty()) {
				aggregate.chiplet_id = chiplet_id;
				aggregate.layer_name = workload["layer_name"].asString();
				aggregate.layer_type = workload["layer_type"].asString();
				aggregate.start_time = meta.start_time;
				aggregate.duration = meta.duration;
			}
			aggregate.members.push_back(&workload_meta.at(workload_id));
			merge_box(aggregate.ofmap_box, workload_ofmap_box(workload));
			merge_box(aggregate.ifmap_box, workload_ifmap_box(workload));
			aggregate.exact_ofmap_elements += box_elements(workload_ofmap_box(workload));
		}
	}

	for(const auto& [key, aggregate] : compute_aggregates) {
		(void)key;
		Json::Value payload;
		payload["type"] = "compute";
		payload["chiplet_id"] = aggregate.chiplet_id;
		payload["time_start"] = json_u64(aggregate.start_time);
		payload["time_end"] = json_u64(aggregate.start_time + aggregate.duration);
		payload["duration"] = json_u64(aggregate.duration);
		payload["layer_name"] = aggregate.layer_name;
		payload["layer_type"] = aggregate.layer_type;
		payload["partition_count"] = static_cast<Json::Value::UInt>(aggregate.members.size());
		payload["fragmented"] = aggregate.exact_ofmap_elements != box_elements(aggregate.ofmap_box);
		payload["ofmap_elements"] = json_u64(aggregate.exact_ofmap_elements);
		payload["ofmap_bbox_elements"] = json_u64(box_elements(aggregate.ofmap_box));
		payload["ofmap"] = box_to_json(aggregate.ofmap_box);
		payload["ifmap"] = box_to_json(aggregate.ifmap_box);
		for(const WorkloadMeta* member : aggregate.members) {
			payload["workload_ids"].append(member->workload_id);
			payload["core_ids"].append(member->core_id);
		}

		auto it = layer_index.find(aggregate.layer_name);
		if(it != layer_index.end()) {
			payload["layer_shape"] = layer_shape_json(network->getNode(it->second).layer());
		}
		chiplet_events[aggregate.chiplet_id].push_back({aggregate.start_time, 1, payload});
	}

	const Json::Value& dram = ir["-1"];
	if(dram.isMember("out")) {
		for(const Json::Value& dram_out : dram["out"]) {
			const std::string tensor_type = dram_out["type"].asString() == "weight" ? "weight" : "activation";
			const std::uint64_t transfer_bytes = bits_to_bytes(dram_out["size"]);
			const unsigned int transfer_id = dram_out["transfer_id"].asUInt();
			for(const Json::Value& destination : dram_out["destination"]) {
				if(destination["type"].asString() != "core" || !destination.isMember("workload_id")) {
					continue;
				}
					const auto meta_it = workload_meta.find(destination["workload_id"].asUInt());
					if(meta_it == workload_meta.end()) {
						continue;
					}
					const WorkloadMeta& dst = meta_it->second;
					const std::string dst_layer_name = (*dst.workload)["layer_name"].asString();
					const std::string key = transfer_key(
						dst.chiplet_id, dst.start_time, 0, "read", tensor_type, dst_layer_name, "dram", -1
					);
					auto& aggregate = transfer_aggregates[key];
					aggregate.chiplet_id = dst.chiplet_id;
					aggregate.time = dst.start_time;
					aggregate.type_rank = 0;
					aggregate.event_type = "read";
					aggregate.tensor_type = tensor_type;
					aggregate.layer_name = dst_layer_name;
				aggregate.peer_kind = "dram";
				aggregate.peer_chiplet_id = -1;
				aggregate.bytes += transfer_bytes;
				aggregate.transfer_ids.insert(transfer_id);
				aggregate.workload_ids.insert(dst.workload_id);
			}
		}
	}

	for(const auto& [workload_id, meta] : workload_meta) {
		(void)workload_id;
		const Json::Value& workload = *meta.workload;
		const std::string layer_name = workload["layer_name"].asString();
		if(workload.isMember("ofmap")) {
			for(const Json::Value& ofmap : workload["ofmap"]) {
				if(!ofmap.isMember("destination")) {
					continue;
				}
				const std::uint64_t transfer_bytes = bits_to_bytes(ofmap["size"]);
				const unsigned int transfer_id = ofmap["transfer_id"].asUInt();
				for(const Json::Value& destination : ofmap["destination"]) {
					const std::string destination_type = destination["type"].asString();
					std::string peer_kind;
					int peer_chiplet_id = -1;
					if(destination_type == "DRAM") {
						peer_kind = "dram";
					} else if(destination_type == "core") {
						peer_kind = "chiplet";
						peer_chiplet_id = core_to_chiplet(destination["id"].asInt()).chiplet_id;
						if(peer_chiplet_id == meta.chiplet_id) {
							continue;
						}
					} else {
						continue;
					}
					const std::string key = transfer_key(
						meta.chiplet_id, meta.end_time, 2, "write", "ofmap", layer_name, peer_kind, peer_chiplet_id
					);
					auto& aggregate = transfer_aggregates[key];
					aggregate.chiplet_id = meta.chiplet_id;
					aggregate.time = meta.end_time;
					aggregate.type_rank = 2;
					aggregate.event_type = "write";
					aggregate.tensor_type = "ofmap";
					aggregate.layer_name = layer_name;
					aggregate.peer_kind = peer_kind;
					aggregate.peer_chiplet_id = peer_chiplet_id;
					aggregate.bytes += transfer_bytes;
					aggregate.transfer_ids.insert(transfer_id);
					aggregate.workload_ids.insert(meta.workload_id);

					if(destination_type != "core" || !destination.isMember("workload_id")) {
						continue;
					}

					const auto dst_meta_it = workload_meta.find(destination["workload_id"].asUInt());
					if(dst_meta_it == workload_meta.end()) {
						continue;
					}
					const WorkloadMeta& dst = dst_meta_it->second;
					const std::string dst_layer_name = (*dst.workload)["layer_name"].asString();
					const std::string read_key = transfer_key(
						dst.chiplet_id, dst.start_time, 0, "read", "activation", dst_layer_name, "chiplet", meta.chiplet_id
					);
					auto& read_aggregate = transfer_aggregates[read_key];
					read_aggregate.chiplet_id = dst.chiplet_id;
					read_aggregate.time = dst.start_time;
					read_aggregate.type_rank = 0;
					read_aggregate.event_type = "read";
					read_aggregate.tensor_type = "activation";
					read_aggregate.layer_name = dst_layer_name;
					read_aggregate.peer_kind = "chiplet";
					read_aggregate.peer_chiplet_id = meta.chiplet_id;
					read_aggregate.bytes += transfer_bytes;
					read_aggregate.transfer_ids.insert(transfer_id);
					read_aggregate.workload_ids.insert(dst.workload_id);
				}
			}
		}
	}

	for(const auto& [key, aggregate] : transfer_aggregates) {
		(void)key;
		Json::Value payload;
		payload["type"] = aggregate.event_type;
		payload["chiplet_id"] = aggregate.chiplet_id;
		payload["time"] = json_u64(aggregate.time);
		payload["layer_name"] = aggregate.layer_name;
		payload["tensor_type"] = aggregate.tensor_type;
		payload["bytes"] = json_u64(aggregate.bytes);
		payload["peer"]["kind"] = aggregate.peer_kind;
		if(aggregate.peer_kind == "chiplet") {
			payload["peer"]["chiplet_id"] = aggregate.peer_chiplet_id;
		}
		for(unsigned int transfer_id : aggregate.transfer_ids) {
			payload["transfer_ids"].append(transfer_id);
		}
		for(unsigned int workload_id : aggregate.workload_ids) {
			payload["workload_ids"].append(workload_id);
		}
		chiplet_events[aggregate.chiplet_id].push_back({aggregate.time, aggregate.type_rank, payload});
	}

	std::vector<ScalesimRow> csv_rows;
	for(const auto& [key, aggregate] : compute_aggregates) {
		(void)key;
		auto it = layer_index.find(aggregate.layer_name);
		if(it == layer_index.end()) {
			continue;
		}
		const Layer& layer = network->getNode(it->second).layer();
		if(!(REF_IS_INSTANCE(layer, ConvLayer) || REF_IS_INSTANCE(layer, GroupConvLayer) || REF_IS_INSTANCE(layer, FCLayer))) {
			continue;
		}
		auto rows = csv_rows_from_aggregate(aggregate, layer);
		csv_rows.insert(csv_rows.end(), rows.begin(), rows.end());
	}
	std::sort(csv_rows.begin(), csv_rows.end(), [](const ScalesimRow& lhs, const ScalesimRow& rhs) {
		return std::tie(lhs.chiplet_id, lhs.start_time, lhs.row_name) <
			std::tie(rhs.chiplet_id, rhs.start_time, rhs.row_name);
	});

	Json::Value chiplet_json;
	chiplet_json["metadata"]["network_name"] = network_name;
	chiplet_json["metadata"]["xlen"] = Cluster::xlen;
	chiplet_json["metadata"]["ylen"] = Cluster::ylen;
	chiplet_json["metadata"]["x_cut"] = NoC::x_cut;
	chiplet_json["metadata"]["y_cut"] = NoC::y_cut;
	chiplet_json["metadata"]["x_step"] = NoC::x_step;
	chiplet_json["metadata"]["y_step"] = NoC::y_step;
	chiplet_json["metadata"]["chiplet_count"] = NoC::x_cut * NoC::y_cut;
	chiplet_json["metadata"]["source"] = "Gemini per-core IR";
	chiplet_json["metadata"]["event_order"] = "pairable_per_core_reconstructed";

	auto ordered_chiplet_events = build_pairable_chiplet_events(chiplet_events, workload_meta);

	for(int chiplet_y = 0; chiplet_y < NoC::y_cut; ++chiplet_y) {
		for(int chiplet_x = 0; chiplet_x < NoC::x_cut; ++chiplet_x) {
			const int chiplet_id = chiplet_y * static_cast<int>(NoC::x_cut) + chiplet_x;
			Json::Value chiplet_entry;
			chiplet_entry["chiplet_id"] = chiplet_id;
			chiplet_entry["chiplet_x"] = chiplet_x;
			chiplet_entry["chiplet_y"] = chiplet_y;
			for(int core_id : chiplet_core_ids[chiplet_id]) {
				chiplet_entry["core_ids"].append(core_id);
			}
			auto& events = ordered_chiplet_events[chiplet_id];
			for(size_t event_index = 0; event_index < events.size(); ++event_index) {
				events[event_index].payload["seq"] = static_cast<Json::Value::UInt>(event_index);
				chiplet_entry["events"].append(events[event_index].payload);
			}
			chiplet_json["chiplets"].append(chiplet_entry);
		}
	}

	const std::filesystem::path output_dir = output_dir_from_hint(output_hint_path);
	std::filesystem::create_directories(output_dir);

	ExportArtifacts artifacts;
	artifacts.chiplet_events_path = (output_dir / "chiplet_events.json").string();
	artifacts.scalesim_topology_path = (output_dir / "scalesim_topology.csv").string();
	artifacts.chiplet_timeline_path = (output_dir / "chiplet_timeline.txt").string();

	Json::StyledWriter writer;
	std::ofstream json_ofs(artifacts.chiplet_events_path);
	json_ofs << writer.write(chiplet_json);
	write_scalesim_csv(artifacts.scalesim_topology_path, csv_rows);
	write_chiplet_timeline_text(artifacts.chiplet_timeline_path, chiplet_json);
	return artifacts;
}
