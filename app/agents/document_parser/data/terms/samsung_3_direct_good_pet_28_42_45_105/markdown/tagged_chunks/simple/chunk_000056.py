from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 평균공시이율+1%를 연단위\n'
 '- 복리로 계산한 금액을 더하여 지급합니다. 다만, 회사는 계약자가 제1회 보험료를 신\n'
 '- 용카드로 납입한 계약의 승낙을 거절하는 경우에는 신용카드의 매출을 취소하며 이자\n'
 '- 를 더하여 지급하지 않습니다.\n'
 '- ⑤ 회사가 제2항에 따라 일부보장 제외 조건을 붙여 승낙하더라도 청약일로부터 5년이\n'
 '- 지나는 동안 보장이 제외되는 질병으로 추가 진단(단순 건강검진 제외) 또는 치료 사'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000056',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
