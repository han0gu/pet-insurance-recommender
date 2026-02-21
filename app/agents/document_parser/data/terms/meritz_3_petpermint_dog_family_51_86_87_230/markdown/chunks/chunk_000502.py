from langchain_core.documents import Document

chunk = Document(
    page_content=('니다.- ① 피보험자가 피해자에 대하여 부담하는 법률상의 손해\n'
 '- 배상책임액이 보험증권에 기재된 보상한도액을 명백하\n'
 '- 게 초과하는 때\n'
 '- ② 피보험자가 정당한 이유없이 협력하지 않은 때\n'
 '\uf000 회사가 제1항의 절차를 대행하는 경우에는, 피보험자에\n'
 '대하여 보상책임을 지는 한도 내에서, 가압류나 가집행을\n'
 '면하기 위한 공탁금을 피보험자에게 대부할 수 있으며 이에\n'
 '소요되는 비용을 보상합니다. 이 경우 대부금의 이자는 공\n'
 '탁금에 붙여지는 것과 같은 이율로 하며, 피보험자는 공탁\n'
 '금(이자를 포함합니다)의 회수청구권을 회사에 양도하여야'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
