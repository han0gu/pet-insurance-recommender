from langchain_core.documents import Document

chunk = Document(
    page_content=('이 법에서 사용하는 용어의 뜻은 다음과 같다. 1. "수의사"란 수의업무를 담당하는 사람으로서 농림축산식 품부장관의 면허를 받은 사람을 '
 '말 한다. 2. "동물"이란 소, 말, 돼지, 양, 개, 토끼, 고양이, 조류(鳥類), 꿀벌, 수생동물(水生動物), 그 밖 에 '
 '대통령령으로 정하는 동물을 말한다. 3. "동물진료업"이란 동물을 진료[동물의 사체 검안(檢案)을 포함한다. 이하 같다]하거나 동물 의 '
 '질병을 예방하는 업(業)을 말한다. 3의2'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000565',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
