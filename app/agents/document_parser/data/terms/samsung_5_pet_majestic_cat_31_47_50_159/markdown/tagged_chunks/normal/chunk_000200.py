from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하고자 할 경우에는 계약자에게 서면 또는 「전자서명법」 제2조 제2호에 따른 전자\n'
 '- 서명으로 동의를 얻어 수신확인을 조건으로 전자문서를 송신하여야 합니다. 계약자의\n'
 '- 전자문서 수신이 확인되기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다. 회\n'
 '- 사는 전자문서가 수신되지 않은 것을 확인한 경우에는 서면(등기우편 등)으로 다시 알\n'
 '- 려드립니다.\n'
 '- ⑤ 제1항 제2호에 의한 특별약관의 해지가 보험금 지급사유 발생 및 보험료 납입면제사'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000200',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
