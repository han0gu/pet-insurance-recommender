from langchain_core.documents import Document

chunk = Document(
    page_content=('- 흡인(吸引): 주사기 등으로 빨아들이는 것\n'
 '- 천자(穿刺): 바늘 또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것제5조(보험금을 지급하지 않는 사유)① 회사는 다음 중 '
 '어느 한 가지로 보험금 지급사유가 발생한 때에는 보험금을 지급하지\n'
 '않습니다.1. 계약자, 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실【중과실(중대한 과실)】주의의무의 위반이 현저한 '
 '과실,「중대한 과실」, 즉 현저한 부주의, 태만의 경우\n'
 '로서 조금만 주의를 하였다면 충분히 피해의 발생을 막을 수 있었음에도 그 주의'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
