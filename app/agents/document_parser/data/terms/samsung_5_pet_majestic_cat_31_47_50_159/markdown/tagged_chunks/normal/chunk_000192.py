from langchain_core.documents import Document

chunk = Document(
    page_content=('- ∙ 상해사망 보험요율 : 1급 0.3, 2급 0.5\n'
 '- → 고객이 수령하는 상해사망 보험금 = 1억원 × (0.3 ÷ 0.5) = 6천만원\n'
 '⑤ 계약자 또는 피보험자가 고의 또는 중대한 과실로 제1항 각 호의 변경사실을 회사에\n'
 '알리지 않았을 경우 변경후 요율이 변경전 요율보다 높을 때에는 회사는 그 변경사실\n'
 '을 안 날부터 1개월 이내에 계약자 또는 피보험자에게 제4항에 따라 보장됨을 통보하\n'
 '고 이에 따라 보험금을 지급합니다.<유의사항>계약자 또는 피보험자는 상해보험계약을 맺은 후 피보험자가 직업 또는 직무를 '
 '변경(자가용운전자가'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000192',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
