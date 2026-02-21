from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 증거자료의 제출을 요구할 수 있습니다.\n'
 '# 제7조(보험금의 청구)① 피보험자가 보험금을 청구할 때에는 다음의 서류를 회사에 제출하여야 합니다.- 1. 보험금 청구서(회사 '
 '양식)\n'
 '- 2. 등록견의 경우에는 동물등록증 또는 등록번호\n'
 '- 3. 미등록견의 경우에는 가입동물의 사진 2매(얼굴전면, 측면전신사진)를 회사에 제출하시고 가입\n'
 '- 동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다.\n'
 '- 4. 진료비 내역서(진료항목이 기재되어 있는 명세서, 수의사 처방전 포함) 및 치료비 영수증'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
