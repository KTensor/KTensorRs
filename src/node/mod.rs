/// Computational Node for a Graph
pub struct Node {
    nodeid: &'static str
}

impl Node {
    pub fn get_id(&self) -> &'static str {
        self.nodeid
    }
}
