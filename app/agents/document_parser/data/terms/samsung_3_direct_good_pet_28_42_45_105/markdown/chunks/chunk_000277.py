from langchain_core.documents import Document

chunk = Document(
    page_content=('니다) 중에 질병으로 사망한 경우 5년간 매월 보험증권에 기재된 이 특별약관의 보험가\n'
 '입금액을 보험금 지급사유 발생일(단, 해당월에 보험금 지급사유 발생일이 없는 경우에는\n'
 '해당월의 마지막 날로 합니다)에 반려동물 양육자금Ⅱ으로 보험수익자에게 지급합니다.# 제 2조 (보험금 지급에 관한 세부규정)- ① '
 '「호스피스∙완화의료 및 임종과정에 있는 환자의 연명의료결정에 관한 법률」에 따른\n'
 '- 연명의료중단 등 결정 및 그 이행으로 피보험자가 사망하는 경우 연명의료중단 등 결'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
