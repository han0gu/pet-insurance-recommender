from langchain_core.documents import Document

chunk = Document(
    page_content=('- 문구와 함께 계약자에게 서면 또는 전자문서 등으로 알려 드립니다.\n'
 '- 회사가 전자문서로 안내하고자 할 경우에는 계약자에게 서면 또는 전자서명법 제2\n'
 '- 조 제2호에 따른 전자서명으로 동의를 얻어 수신확인을 조건으로 전자문서를 송신\n'
 '- 하여야 합니다. 계약자의 전자문서 수신이 확인되기 전까지는 그 전자문서는 송신\n'
 '- 되지 않은 것으로 봅니다. 회사는 전자문서가 수신되지 않은 것을 확인한 경우에\n'
 '- 는 서면(등기우편 등)으로 다시 알려드립니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000703',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
