#ifndef EXPORT_H
#define EXPORT_H

#include <string>

class SchNode;

struct ExportArtifacts {
	std::string chiplet_events_path;
	std::string scalesim_topology_path;
	std::string chiplet_timeline_path;
};

ExportArtifacts export_chiplet_artifacts(
	const SchNode& schedule,
	const std::string& output_hint_path,
	const std::string& network_name
);

#endif // EXPORT_H
