from langchain_core.documents import Document

chunk = Document(
    page_content=('- 제1항에 따른 납입최고(독촉) 등을 실시할 것\n'
 '- 4. 전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 있는 기능을 갖출 것\n'
 '- 5. 제3호 및 제4호의 내용에 관한 사항을 계약자에게 안내할 것\n'
 '- \uf000 제1항에 따라 계약이 해지된 경우에는 제34조(해약환급금) 제1항에 따른 해약환급\n'
 '금을 계약자에게 지급합니다.\n'
 '용 어 풀 이 납입최고(독촉)| 약정된 기일까지 | 보험료가 납입되지 |  | 않을 경우, 회사가 계약자에게 보험료의 |\n'
 '| --- | --- | --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000165',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
