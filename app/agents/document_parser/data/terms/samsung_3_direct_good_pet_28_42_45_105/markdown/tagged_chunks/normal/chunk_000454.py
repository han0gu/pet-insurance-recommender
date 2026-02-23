from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. "동물"이란 소, 말, 돼지, 양, 개, 토끼, 고양이, 조류(鳥類), 꿀벌, 수생동물(水生動物), 그 밖에 대\n'
 '- 통령령으로 정하는 동물을 말한다.\n'
 '- 3. "동물진료업"이란 동물을 진료[동물의 사체 검안(檢案)을 포함한다. 이하 같다]하거나 동물의 질\n'
 '- 병을 예방하는 업(業)을 말한다.\n'
 '- 3의2. "동물보건사"란 동물병원 내에서 수의사의 지도 아래 동물의 간호 또는 진료 보조 업무에 종\n'
 '- 사하는 사람으로서 농림축산식품부장관의 자격인정을 받은 사람을 말한다.'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000454',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
