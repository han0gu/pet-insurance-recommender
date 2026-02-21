from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 종사하는 사람으로서 농림축산식품부장관의 자격인정을 받은 사람을 말한다.\n'
 '- 4. "동물병원"이란 동물진료업을 하는 장소로서 제17조에 따른 신고를 한 진료기관을 말한다.\n'
 '③ 제1항 제4호의 사고증명서는 수의사법 제12조(진단서 등)에서 규정한 내용에 따라 국\n'
 '내의 동물병원에서 수의사에 의해 발급한 것이어야 합니다.<수의사법 제12조(진단서 등)>- ① 수의사는 자기가 직접 진료하거나 검안하지 '
 '아니하고는 진단서, 검안서, 증명서 또는 처방'),
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
 'indexing': {'chunk_id': 'chunk_000322',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
