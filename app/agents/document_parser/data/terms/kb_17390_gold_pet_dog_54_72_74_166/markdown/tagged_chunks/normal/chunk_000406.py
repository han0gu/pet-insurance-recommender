from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 "호흡기관련질병"의 진단확정은 의료법 제3조에서 정한 국내의 병원이나 의원 또\n'
 '는 국외의 의료관련법에서 정한 의료기관의 의사자격을 가진 자에 의한 진단서에\n'
 '의합니다. 또한 회사가 "호흡기관련질병"의 조사나 확인을 위하여 필요하다고 인\n'
 '정하는 경우 검사결과, 진료기록부의 사본제출을 요청할 수 있습니다.- \n'
 '# 제4조(수술의 정의와 장소)- \uf000 이 특별약관에 있어서 "수술"이라 함은 병원 또는 의원의 의사, 치과의사 면허를\n'
 '- 가진 자(이하 "의사"라 합니다)에 의하여 치료가 필요하다고 인정한 경우로서 자'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000406',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
