from langchain_core.documents import Document

chunk = Document(
    page_content=("제3조(의료기관) 제2항에서 정한 국내의 병</p><br><table id='240' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>원이나 의원 "
 '또는</td><td>국외의</td><td>정한 의료기관을 말합니다.</td><td>의료관련법에서</td></tr><tr><td '
 'colspan="4">용 어 풀 이 ∙ 절단 : 특정부위를 잘라 내는 것 ∙ 절제 : 특정부위를 잘라 없애는 것 ∙ 흡인 : 주사기 '
 '등으로 빨아들이는 것 ∙ 천자 : 바늘 또는 관을 꽂아 체액․조직을 뽑아내거나'),
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
