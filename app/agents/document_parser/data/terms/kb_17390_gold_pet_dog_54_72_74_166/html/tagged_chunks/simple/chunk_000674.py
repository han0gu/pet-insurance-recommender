from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또한 회사가 "호흡기관련질병"의 조사나 확인을 위하여 필요하다고 인<br>정하는 경우 검사결과, 진료기록부의 사본제출을 요청할 수 '
 "있습니다.</p><h1 id='231' style='font-size:14px'>제4조(수술의 정의와 장소)</h1><br><p "
 "id='232' data-category='list' style='font-size:14px'>\uf000 이 특별약관에 있어서 "
 '"수술"이라 함은 병원 또는 의원의 의사, 치과의사 면허를<br>가진 자(이하 "의사"라 합니다)에 의하여 치료가 필요하다고 인정한 '
 '경우로서 자<br>택'),
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
 'indexing': {'chunk_id': 'chunk_000674',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
