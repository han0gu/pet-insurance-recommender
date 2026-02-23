from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 피보험자가 피해자에 대하여 부담하는 법률상의 손해배상책임액이 보험증권에 기재\n'
 '- 된 보상한도액을 명백하게 초과하는 때\n'
 '- 2. 피보험자가 정당한 이유없이 협력하지 않은 때\n'
 '⑤ 회사가 제1항의 절차를 대행하는 경우에는, 피보험자에 대하여 보상책임을 지는 한도\n'
 '내에서, 가압류나 가집행을 면하기 위한 공탁금을 피보험자에게 대부할 수 있으며 이에\n'
 '소요되는 비용을 보상합니다. 이 경우 대부금의 이자는 공탁금에 붙여지는 것과 같은\n'
 '이율로 하며, 피보험자는 공탁금(이자를 포함합니다)의 회수청구권을 회사에 양도하여'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000147',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
