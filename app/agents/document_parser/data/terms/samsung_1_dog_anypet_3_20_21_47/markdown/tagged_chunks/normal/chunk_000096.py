from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하는 경우에는 적용하지 않습니다.\n'
 "【수술】 동물병원의 수의사 자격을 가진 자(이하 '수의사'라 합니다)에 의하여 치료가 필요하다고 인정된\n"
 '상해 또는 질병 치료를 위하여 수의사법 제17조(개설)에서 규정한 국내의 동물병원에서 수의사의 관리 하\n'
 '에 직접적인 치료를 목적으로 기구를 사용하여 생체에 절개, 절단, 절제 등의 조작을 가하는 것을 말합니\n'
 '다. 단 수술에서 아래에 정한 사항은 제외합니다.- 1. 흡인 (주사기 등으로 빨아 들이는 것)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000096',
              'chunk_char_len': 248,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
