from mcp.server.fastmcp import FastMCP
from pydantic import Field
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("DocumentMCP", log_level="ERROR")


docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}

# tools

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

# resources

@mcp.resource(
    "docs://documents/",
    mime_type="application/json"
)
def list_documents() -> list[str]:
    return list(docs.keys());

@mcp.resource(
    "docs://documents/{doc_id}/",
    mime_type="text/plain"
)
def fetch_doc(doc_id:str) -> str:
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")
    
    return docs[doc_id]

# prompts
@mcp.prompt(
    name="format",
    description="Rewrites the contents of the document in markdown format"
)
def format_doc(
    doc_id:str=Field(description="The doc id of the document to be formatted")
) -> list[base.Message]:
    prompt = f"""
        Your goal is to reformat a document to be written with markdown syntax.

        The id of the document you need to reformat is:
        <document_id>
        {doc_id}
        </document_id>

        Add in headers, bullet points, tables, etc as necessary. Feel free to add in structure.
        Use the 'edit_document' tool to edit the document. After the document has been reformatted...
    """
    
    return [base.UserMessage(prompt)]


if __name__ == "__main__":
    mcp.run(transport="stdio")
