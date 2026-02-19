from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 제3항에도 불구하고 전화를 이용하여 계약을 체결하는 경우 다음의 각 호의 어느 하나 를 충족하는 때에는 자필서명을 생략할 수 '
 '있으며, 제2항의 규정에 따른 음성녹음 내용 을 문서화한 확인서를 계약자에게 드림으로써 계약자 보관용 청약서를 전달한 것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 13},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000083',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
