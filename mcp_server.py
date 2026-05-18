from mcp.server.fastmcp import FastMCP
from pydantic import Field
mcp = FastMCP("DocumentMCP", log_level="ERROR")


docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}

@mcp.tool(
    name="read_doc",
    description="Reads the content of the provided doc id"
)
def read_doc(
    doc_id:str = Field(description="The id of the document to be read and it's of type string")
) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document {doc_id} not found")
    
    return docs[doc_id]

@mcp.tool(
    name="edit_doc",
    description="Replace an old string with a new one in a doc, must match exactly including whitespaces"
)
def edit_doc(
    doc_id:str = Field(description="The id of the document to edit"),
    old_string:str = Field(description="The old string to replace"),
    new_string:str = Field(description="The new value of the string to replace the old one")
):
    if doc_id not in docs:
        raise ValueError(f"Document {doc_id} cannot be found")
    
    docs[doc_id] = docs[doc_id].replace(old_string, new_string)



if __name__ == "__main__":
    mcp.run(transport="stdio")
