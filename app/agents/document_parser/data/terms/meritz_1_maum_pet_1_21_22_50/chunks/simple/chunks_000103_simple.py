from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 있는 기능을 갖출 것 5. 제3호 및 제4호의 내용에 관한 사항을 계약자에게 '
 '안내할 것\n'
 '④ 제1항에 따라 계약이 해지되고 이로 인하여 회사가 환급하여야 할 보험료가 있을 때에 는 제33조(보험료의 환급)에 따른 보험료를 '
 '계약자에게 지급합니다.\n'
 '【납입최고(독촉)】\n'
 '약정된 기일까지 보험료가 납입되지 않을 경우, 회사가 계약자에게 납입을 재촉하 는 것을 말합니다.\n'
 '제28조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 16},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000103',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
