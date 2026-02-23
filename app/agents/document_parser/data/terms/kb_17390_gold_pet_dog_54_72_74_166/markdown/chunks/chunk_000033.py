from langchain_core.documents import Document

chunk = Document(
    page_content=('| 임을 물을 만한 기대가능성이 없을 때)를 말하며, 가령 천재지변, 전쟁, 사변 등으로 인해 이행이 불가능한 경우 등이 있습니다. '
 '\uf000 회사는 제6항의 서면조사에 대한 동의 요청시 조사목적, 사용처 등을 명시하고 설 | 임을 물을 만한 기대가능성이 없을 때)를 '
 '말하며, 가령 천재지변, 전쟁, 사변 등으로 인해 이행이 불가능한 경우 등이 있습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
