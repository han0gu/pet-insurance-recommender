from langchain_core.documents import Document

chunk = Document(
    page_content=('합니다) 중에 상해를 입고 병원 또는 의원(한방병원 또는 한의원을 포함합니다) 등에\n'
 '서 치료를 받고 그직접적인 결과로써 안면부, 상지, 하지에 외형상의 반흔(흉터)이나\n'
 '추상(추한 모습)장해, 신체의 기형이나 기능장해가 발생하여 그 원상회복(이하 「상해\n'
 '흉터복원」 이라 합니다)을 목적으로 사고일부터 2년 이내에 성형외과 전문의로부터\n'
 '성형수술(단, 사고발생시점 만15세 미만자의 경우 부득이 사고일부터 2년이 지난 후\n'
 '에 성형수술이 가능하다는 진단을 받은 경우에는 그 진단으로 대체할 수 있습니다)을'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000355',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
