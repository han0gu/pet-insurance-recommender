from langchain_core.documents import Document

chunk = Document(
    page_content=('- 물\n'
 '- 받은 사람을 말한다.\n'
 '- 2. "동물"이란 소, 말, 돼지, 양, 개, 토끼, 고양이, 조류(鳥類), 꿀벌, 수생\n'
 '- 동물(水生動物), 그 밖에 대통령령으로 정하는 동물을 말한다.\n'
 '- 제\n'
 '- 3. "동물진료업"이란 동물을 진료[동물의 사체 검안(檢案)을 포함한다. 이하\n'
 '- 도\n'
 '- 같다]하거나 동물의 질병을 예방하는 업(業)을 말한다.\n'
 '- 성\n'
 '- 4. "동물병원"이란 동물진료업을 하는 장소로서 제17조에 따른 신고를 한 진료\n'
 '- 특\n'
 '- 기관을 말한다.\n'
 '- 약'),
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
 'indexing': {'chunk_id': 'chunk_000614',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
