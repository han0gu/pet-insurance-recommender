from langchain_core.documents import Document

chunk = Document(
    page_content=('재진단 또는 치료를 받은 것으로 간주합니다.# ⑤ 제4항의 재진단 또는 치료를 받지 않은 경우는 다음 각 호의 경우를 포함합니다.- 1. '
 '검진결과 추가검사 또는 치료가 필요하지 않았던 경우\n'
 '- 2. 제1항 제1호에서 정한 특정신체부위에 발생한 질병 또는 제1항 제2호에서 정한 특\n'
 '- 정질병이 악화되지 않고 유지된 경우\n'
 '⑥ 제1항의 규정에도 불구하고 다음 사항 중 어느 한 가지의 경우에 해당되는 사유로 보\n'
 '험계약에서 정한 보험금의 지급사유가 발생한 경우 회사는 보험금을 지급하여 드리며,'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000565',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
