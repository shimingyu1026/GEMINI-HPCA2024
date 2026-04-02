#include "export.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
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
			auto& events = chiplet_events[chiplet_id];
			std::sort(events.begin(), events.end(), [](const GenericEvent& lhs, const GenericEvent& rhs) {
				return std::tie(lhs.sort_time, lhs.type_rank) < std::tie(rhs.sort_time, rhs.type_rank);
			});
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

	Json::StyledWriter writer;
	std::ofstream json_ofs(artifacts.chiplet_events_path);
	json_ofs << writer.write(chiplet_json);
	write_scalesim_csv(artifacts.scalesim_topology_path, csv_rows);
	return artifacts;
}
