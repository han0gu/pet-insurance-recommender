from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 제1항의 청구를 받았을 때에는 지체없이 피보험자에게 통지하여야 하며, 회사 의 요구가 있으면 계약자 및 피보험자는 필요한 서류 '
 '· 증거의 제출, 증언 또는 증인 출석에 협조하여야 합니다. ③ 피보험자가 피해자로부터 손해배상의 청구를 받았을 경우에 회사가 필요하다고 '
 '인정 할 때에는 피보험자를 대신하여 회사의 비용으로 이를 해결할 수 있습니다. 이 경우 회사의 요구가 있으면 계약자 또는 피보험자는 이에 '
 '협력하여야 합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 89},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000577',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
