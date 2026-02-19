from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는 계약자가 제1회 보험료를 신 용카드로 납입한 계약의 승낙을 거절하는 경우에는 신용카드의 매출을 취소하며 이자 를 더하여 '
 '지급하지 않습니다. ⑤ 회사가 제2항에 따라 일부보장 제외 조건을 붙여 승낙하더라도 청약일로부터 5년이 지나는 동안 보장이 제외되는 '
 '질병으로 추가 진단(단순 건강검진 제외) 또는 치료 사 실이 없을 경우, 청약일로부터 5년이 지난 이후에는 이 약관에 따라 보장합니다. '
 '⑥ 제5항의 추가 진단(단순 건강검진 제외) 또는 치료 사실이 없는 경우는 다음 각 호의 경우를 포함합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 34},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000071',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
