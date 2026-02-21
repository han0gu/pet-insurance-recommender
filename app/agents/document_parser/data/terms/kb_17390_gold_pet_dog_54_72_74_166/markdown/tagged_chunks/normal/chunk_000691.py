from langchain_core.documents import Document

chunk = Document(
    page_content=('한도 내에서, 가압류나 가집행을 면하기 위한 공탁금을 피보험자에게 대부할 수\n'
 '있으며 이에 소요되는 비용을 보상합니다. 이 경우 대부금의 이자는 공탁금에 붙\n'
 '여지는 것과 같은 이율로 하며, 피보험자는 공탁금(이자를 포함합니다)의 회수청 반\n'
 '구권을 회사에 양도하여야 합니다. 려동\n'
 '용 어 풀 이 보상책임을 지는 한도| 동일한 사고로 이미 지급한 | 보험금이나 가지급보험금이 |  | 물 있는 경우에는 그 금액 |\n'
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
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000691',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
