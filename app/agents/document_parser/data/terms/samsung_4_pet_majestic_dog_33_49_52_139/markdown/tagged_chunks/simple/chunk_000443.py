from langchain_core.documents import Document

chunk = Document(
    page_content=('정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약자적립액 및 미경과보험료\n'
 '를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없습니다.- 96 -4 특별약관\n'
 '관련\n'
 '面4-1. 반려견 의료비(치과및구강질환포함)(수술당일제외,\n'
 '검 사비포함)(재가입형) 특별 약관# 제1조 (목적)이 특별약관은 보험계약자(이하 「계약자」 라 합니다)와 보험회사(이하 「회사」 라 '
 '합니다)\n'
 '사이에 보험증권에 기재된 반려견의 상해 또는 질병으로 인한 위험을 보장하기 위하여'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000443',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
