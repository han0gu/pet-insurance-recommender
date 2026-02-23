from langchain_core.documents import Document

chunk = Document(
    page_content=('잘라 내는 것 ∙ 절제 : 특정부위를 잘라 없애는 것 ∙ 흡인 : 주사기 등으로 빨아들이는 '
 "것</td></tr></tbody></table><br><h1 id='44' style='font-size:14px'>∙ 천자 : 바늘 "
 "또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것</h1><p id='45' data-category='paragraph' "
 "style='font-size:14px'>제7조(특별약관의 소멸)<br>\uf000 보험증권에 기재된 반려동물이 보험기간 중에 사망하여 "
 '보험의 목적에 대해 이<br>특별약관에서 정한'),
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
