from langchain_core.documents import Document

chunk = Document(
    page_content=('10. 한방 및 한약(보상하는 상해 또는 질병의 치료를 위한 침술 및 물리치료는 제외합 니다.), 온천요법, 산소요법, 면역요법 등의 '
 '대체적 처치에 의한 치료를 위한 비용 11. 가입동물의 이송비, 마이크로칩의 삽입 비용, 안락사를 위한 비용, 장례식비용, 매 장비용 등 '
 '가입동물의 사망 후에 소요된 비용, 각종 증명서류의 작성비용(운송비 포함)\n'
 '12. 펫호텔 비용 또는 위탁료, 산책료, 카운슬링 비용, 상담료, 지도료, 문제행동 교정\n'
 '비용 및 이와 동종의 비용'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 82},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000514',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
