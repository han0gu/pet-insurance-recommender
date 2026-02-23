from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 1 | 위.십이지장 |\n'
 '| 2 | 공장(빈창자), 회장(돌창자), 맹장(충수돌기 포함) |\n'
 '| 3 | 대장(맹장, 직장 제외) |\n'
 '| 4 | 직장 |\n'
 '| 5 | 항문 |\n'
 '| 6 | 간 |\n'
 '| 7 | 담낭(쓸개) 및 담관 |\n'
 '| 8 | 췌장 |\n'
 '| 9 | 비장 |\n'
 '| 10 | 기관, 기관지, 폐, 흉막 및 흉곽(늑골 포함) |\n'
 '| 11 | 코[외비(코 바깥), 비강(코 안) 및 부비강(코 곁굴) 포함] |\n'
 '| 12 | 인두 및 후두(편도 포함) |\n'
 '| 13 | 식도 |'),
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
 'indexing': {'chunk_id': 'chunk_000570',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
