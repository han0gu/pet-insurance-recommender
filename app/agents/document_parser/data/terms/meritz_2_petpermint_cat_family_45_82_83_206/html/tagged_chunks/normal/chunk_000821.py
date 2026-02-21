from langchain_core.documents import Document

chunk = Document(
    page_content=("치료가 곤란하여 동물병원에 입실하여 수의사의 관리<br>하에 치료에 전념하는 것을 말합니다.</p><h1 id='64' "
 "style='font-size:20px'>제5조(MRI,CT 및 내시경처치의 정의)</h1><br><p id='65' "
 "data-category='paragraph' style='font-size:20px'>\uf000 이 특별약관에 있어서 MRI,CT 및 "
 '내시경처치라 함은 자<br>기공명영상(MRI), 전산화단층촬영(CT) 및 내시경처치를 말<br>합니다.<br>\uf000 제1항의 '
 '자기공명영상(MRI)이라 함은 제1조(보험금의'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000821',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
