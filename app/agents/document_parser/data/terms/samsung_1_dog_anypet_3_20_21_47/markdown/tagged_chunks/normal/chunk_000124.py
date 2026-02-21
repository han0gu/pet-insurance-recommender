from langchain_core.documents import Document

chunk = Document(
    page_content=('류나 가집행을 면하기 위한 공탁금을 피보험자에게 대부할 수 있으며 이에 소요되는 비용을 보상\n'
 '합니다. 이 경우 대부금의 이자는 공탁금에 붙여지는 것과 같은 이율로 하며, 피보험자는 공탁금\n'
 '(이자를 포함합니다)의 회수청구권을 회사에 양도하여야 합니다.# 제10조(대위권)① 회사가 보험금을 지급한 때(현물보상한 경우를 '
 '포함합니다)에는 회사는 지급한 보험금의 한도내에\n'
 '서 아래의 권리를 가집니다. 다만, 회사가 보상한 금액이 피보험자가 입은 손해의 일부인 경우에는'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000124',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
