from langchain_core.documents import Document

chunk = Document(
    page_content=('가. 보험가입 당시의 연령이 생후 60일 이하 또는 만 10세를 초과하는 개(犬) 나. 판매점, 브리더 등이 매매(賣買)를 목적으로 '
 '사육ㆍ관리하는 개(犬) 다. 경찰견, 구조견, 군견, 사냥개 등 특수한 목적의 개(犬)(단, 맹도견, 청도견 등 장애인 안내견은 제외) '
 '라. 투견, 경주견 등 흥행을 목적으로 사육·관리하는 개(犬)\n'
 '<용어풀이>\n'
 '[흥행]\n'
 '영리를 목적으로 연극, 영화, 서커스 등을 요금을 받고 대중에게 보여주는 행위를 말합니다.\n'
 '마. 유기동물 보호센터 등에서 사육·관리하는 개(犬)\n'
 '② 지급사유 관련 용어'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 66},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000332',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
