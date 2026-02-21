from langchain_core.documents import Document

chunk = Document(
    page_content=('한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관에서 보장| 하는 골절 해당 여부를 판단합니다. |  |\n'
 '| --- | --- |\n'
 '| 대상이 되는 항목 | 분류번호 |\n'
 '| 두개골 및 안면골의 골절 | S02 |\n'
 '| 머리의 으깸손상 | S07 |\n'
 '| 머리의 상세불명 손상 | S09.9 |\n'
 '| 목의 골절 | S12 |\n'
 '| 늑골, 흉골 및 흉추의 골절 | S22 |\n'
 '| 요추 및 골반의 골절 | S32 |\n'
 '| 어깨 및 위팔의 골절 | S42 |\n'
 '| 아래팔의 골절 손목 및 손부위의 골절 | S52 S62 |'),
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
