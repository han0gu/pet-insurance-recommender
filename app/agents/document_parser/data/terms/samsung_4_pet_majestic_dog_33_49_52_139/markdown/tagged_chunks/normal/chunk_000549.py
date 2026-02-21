from langchain_core.documents import Document

chunk = Document(
    page_content=('에 보험증권에 기재된 반려견이 국내에서 수의사에게 이물 섭취 치료를 목적으로 이\n'
 '물제거(내시경) 또는 이물제거(구토유도약물)를 받은 경우 연간 2회에 한하여 당일 피\n'
 '보험자가 부담한 반려견의 치료에 사용된 비용(각종 할인 및 감면, 사후환급금액 등을\n'
 '제외한 실수납액을 의미합니다. 이하「의료비」라 합니다)에서「자기부담금」및「반\n'
 '려견의료비(치과및구강질환포함)(수술당일제외,검사비 포함)보험금의 1일 한도 」를\n'
 '제외한 금액을 아래에 정한 한도로 제5항에 따라 보험수익자에게 반려견의료비확대보'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000549',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
