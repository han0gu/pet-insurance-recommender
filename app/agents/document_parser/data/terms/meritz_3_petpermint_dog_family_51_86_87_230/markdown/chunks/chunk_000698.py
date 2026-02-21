from langchain_core.documents import Document

chunk = Document(
    page_content=('보행이 불가능한 상태(20%) - 보조기구 없이 독립적인 보행은 가능하나 보행 시 파행(절뚝거림)이 있으며, 난간을 잡지 않 고는 계단을 '
 '오르내리기가 불가능한 상태 또는 평지에서 100m 이상을 걷지 못하는 상태(10%) |'),
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
