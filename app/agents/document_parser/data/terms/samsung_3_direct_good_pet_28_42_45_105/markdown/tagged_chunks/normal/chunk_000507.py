from langchain_core.documents import Document

chunk = Document(
    page_content=('- 합니다) 중에 상해를 입고 그 직접결과로써 생활기능 또는 업무능력에 지장을 가져와\n'
 '- 병원 또는 의원(한방병원 또는 한의원을 포함합니다)에 1일이상 계속 입원하여 치료를\n'
 '- 받은 경우에는 입원기간 동안 보험증권에 기재된 반려견을 수탁기관에 위탁함으로써\n'
 '- 발생한 위탁비용을 반려견 위탁비용으로 보험수익자에게 지급합니다. 다만, 반려견 위\n'
 '- 탁비용의 지급일수는 1회 입원당 180일을 한도로 피보험자의 입원기간을 초과할 수\n'
 '- 없습니다.\n'
 '- ② 제1항의 「수탁기관」 이라 함은 동물보호법 시행규칙 제43조(등록영업의 세부 범위)'),
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
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000507',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
