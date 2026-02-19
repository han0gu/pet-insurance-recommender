from langchain_core.documents import Document

chunk = Document(
    page_content=('1. "수의사"란 수의업무를 담당하는 사람으로서 농림축산식품부장관의 면허를 받은 사람을 말한다. 2. "동물"이란 소, 말, 돼지, 양, '
 '개, 토끼, 고양이, 조류(鳥類), 꿀벌, 수생동물(水生動物), 그 밖에 대 통령령으로 정하는 동물을 말한다. 3. "동물진료업"이란 '
 '동물을 진료[동물의 사체 검안(檢案)을 포함한다. 이하 같다]하거나 동물의 질 병을 예방하는 업(業)을 말한다. 3의2'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 119},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000769',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
