from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항 제2호의 삭감기간은 피보험자의 건강상태, 위험의 종류 및 정도에 따라 5년 이 내로 합니다. 단, 해당계약이 계약자가 '
 '보험기간이 끝나는 날까지 계약을 유지하지 않 는다는 의사표시가 없는 한 별도의 계약사정 없이 갱신되는 계약(이하「갱신계약」이 라 '
 '합니다)인 경우「삭감기간」의 산정은 최초 계약일을 기준으로 합니다. 또한 그 판 단기준은 회사에서 정한 계약사정기준을 따르며, 개개인의 '
 '질병의 상태 등에 대한 의 사의 소견에 따라서 다르게 적용할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 127},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000814',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
