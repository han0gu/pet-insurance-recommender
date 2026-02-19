from langchain_core.documents import Document

chunk = Document(
    page_content=('⑦ 피보험자가 회사가 정한 회사가 보험금을 지급하지 않는 기간의 종료일을 포함하여 계속하여 입원한 경우 그 입원에 대해서는 회사가 '
 '보험금을 지급하지 않는 기간 종료 일의 다음날을 입원의 개시일로 인정하여 보험금을 지급합니다. ⑧ 피보험자에게 보험금의 지급사유 또는 '
 '보험료 납입면제사유가 발생했을 경우, 그 보험 금의 지급사유 또는 보험료 납입면제사유가 특정신체부위 또는 특정질병을 직접적인 원인으로 '
 '발생한 사고인가 아닌가는 의사의 진단서와 의견을 주된 판단자료로 결정합 니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000658',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
