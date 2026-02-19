from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사가 제1항에 따른 납입최고(독촉) 등을 전자문서로 안내하고자 할 경우에는 계약자 에게 서면, ⌜전자서명법⌟ 제2조 제2호에 따른 '
 '전자서명으로 동의를 얻어 수신확인을 조건으로 전자문서를 송신하여야 하며, 계약자가 전자문서에 대하여 수신을 확인하기 전까지는 그 '
 '전자문서는 송신되지 않은 것으로 봅니다. 회사는 전자문서가 수신되지 않 은 것을 확인한 경우에는 제1항에서 정한 내용을 서면(등기우편 '
 '등) 또는 전화(음성녹 음)로 다시 알려 드립니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 16},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000100',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
