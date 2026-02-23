from langchain_core.documents import Document

chunk = Document(
    page_content=('절제(切除): 특정부위를 잘라 없애는 것<br>- 흡인(吸引): 주사기 등으로 빨아들이는 것<br>- 천자(穿刺): 바늘 또는 관을 꽂아 '
 "체액․조직을 뽑아내거<br>나 약물을 주입하는 것</p><br><p id='39' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제1항의「수술」은 자택 등에서의 치료가 곤란하여 동물<br>병원에서 행한 것에 "
 "한합니다.</p><h1 id='40' style='font-size:20px'>제4조(입원의 정의와 장소)</h1><br><p "
 "id='41'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000655',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
