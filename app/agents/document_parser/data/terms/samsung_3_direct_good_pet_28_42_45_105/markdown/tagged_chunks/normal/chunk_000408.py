from langchain_core.documents import Document

chunk = Document(
    page_content=('3. 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용\n'
 '선천성 난청, Achalasia(식도·직장 등의 이완 불능증), 구개열, 동맥관 개존증4. 산후 문제행동, 수유에 따르는 칼슘 부족에 '
 '의한 경련 및 기타 임신ㆍ출산과 관련- 78 -78 / 181| <용어풀이> | <용어풀이> |\n'
 '| --- | --- |\n'
 '| 배꼽허니아 | 복부 내장의 탈장 등으로 인해 배꼽 주변이 부풀어 오르는 증상 |\n'
 '| 파보바이러스 감염증 | 파보바이러스에 감염되어 구토와 설사 등의 증상을 일으킴 |'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000408',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
