from langchain_core.documents import Document

chunk = Document(
    page_content=('| 4년이상 5년미만 | 보험계약에 정한 지급보험금 |  |  |  |  | 80% |\n'
 '- ② 제1항 제2호의 삭감기간은 피보험자의 건강상태, 위험의 종류 및 정도에 따라 5년 이\n'
 '- 내로 합니다. 단, 해당계약이 계약자가 보험기간이 끝나는 날까지 계약을 유지하지 않\n'
 '- 는다는 의사표시가 없는 한 별도의 계약사정 없이 갱신되는 계약(이하「갱신계약」이\n'
 '- 라 합니다)인 경우「삭감기간」의 산정은 최초 계약일을 기준으로 합니다. 또한 그 판\n'
 '- 단기준은 회사에서 정한 계약사정기준을 따르며, 개개인의 질병의 상태 등에 대한 의'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000681',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
