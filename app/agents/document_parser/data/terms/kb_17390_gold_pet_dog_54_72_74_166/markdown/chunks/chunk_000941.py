from langchain_core.documents import Document

chunk = Document(
    page_content=('- 저하되는 것을 말한다.\n'
 '- 나) 치매의 장해평가는 임상적인 증상 뿐 아니라 뇌영상검사(CT 및 MRI,\n'
 '- SPECT등)를 기초로 진단되어져야 하며, 18개월 이상 지속적인 치료\n'
 '- 후 평가한다. 다만, 진단시점에 이미 극심한 치매 또는 심한 치매로\n'
 '- 진행된 경우에는 6개월간 지속적인 치료 후 평가한다.\n'
 '- 다) 치매의 장해평가는 전문의(정신건강의학과, 신경과)에 의한 임상치\n'
 '- 매척도(한국판 Expanded Clinical Dementia Rating) 검사결과에 따\n'
 '- 른다.\n'
 '4) 뇌전증-'),
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
