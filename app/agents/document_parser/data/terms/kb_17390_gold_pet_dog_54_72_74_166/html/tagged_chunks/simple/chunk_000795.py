from langchain_core.documents import Document

chunk = Document(
    page_content=('. 1. "수의사"란 수의업무를 담당하는 사람으로서 농림축산식품부장관의 면허를 받은 사람을 말한다. 2. "동물"이란 소, 말, 돼지, '
 '양, 개, 토끼, 고양이, 조류(鳥類), 꿀벌, 수생 동물(水生動物), 그 밖에 대통령령으로 정하는 동물을 말한다. 3. '
 '"동물진료업"이란 동물을 진료[동물의 사체 검안(檢案)을 포함한다. 이하 같다]하거나 동물의 질병을 예방하는 업(業)을 말한다. 4'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000795',
              'chunk_char_len': 217,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
