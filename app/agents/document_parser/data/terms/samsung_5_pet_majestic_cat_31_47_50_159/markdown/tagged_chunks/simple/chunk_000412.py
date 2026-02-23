from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약관 제36조(해약환급금)을 적용합니다.\n'
 '# 제2관 개별사항# 제1조 (보험금의 지급사유)회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합\n'
 '니다) 중에 질병으로 사망한 경우 5년간 매월 보험증권에 기재된 이 특별약관의 보험가\n'
 '입금액을 보험금 지급사유 발생일(단, 해당월에 보험금 지급사유 발생일이 없는 경우에는'),
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
 'indexing': {'chunk_id': 'chunk_000412',
              'chunk_char_len': 193,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
