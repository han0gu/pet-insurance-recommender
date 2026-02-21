from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자가 성명기입란에 본인의 성명을 기재하고, 날인란에 사인(signature) 또 특별\n'
 '는 도장을 찍는 것을 말합니다. 전자서명법 제2조 제2호에 따른 전자서명을 포\n'
 '약\n'
 '함합니다.용 어 풀 이 약관의 중요한 내용금융소비자 보호에 관한 법률 제19조(설명의무)등에서 정한 다음의 내용을 말\n'
 '합니다.∙∙∙∙∙∙∙∙∙∙- 위험보장사항 및 각각의 보험료\n'
 '- 별\n'
 '- 청약의 철회에 관한 사항(기한ㆍ행사방법ㆍ효과 등)\n'
 '- 표\n'
 '- 지급한도, 면책사항, 감액지급 사항 등 보험금 지급제한 조건\n'
 '- 고지의무 및 통지의무 위반의 효과'),
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
