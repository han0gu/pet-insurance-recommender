from langchain_core.documents import Document

chunk = Document(
    page_content=('<유의사항>\n'
 '[수술]\n'
 "동물병원의 수의사 자격을 가진 자(이하 '수의사'라 합니다)에 의하여 치료가 필요하다고 인정된 상 해 또는 질병 치료를 위하여 수의사법 "
 '제 17조(개설)에서 규정한 국내의 동물병원에서 수의사의 관리 하에 직접적인 치료를 목적으로 기구를 사용하여 생체에 절개, 절단, 절제 '
 '등의 조작을 가하\n'
 '는 것을 말합니다. 단 수술에서 아래에 정한 사항은 제외합니다\n'
 '1. 흡인 (주사기 등으로 빨아 들이는 것)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 79},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000479',
              'chunk_char_len': 233,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
