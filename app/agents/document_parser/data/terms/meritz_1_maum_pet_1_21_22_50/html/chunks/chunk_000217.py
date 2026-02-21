from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자의 지시에 따른 배상책임<br>11. 벌과금 및 징벌적 손해에 대한 배상책임<br>12. 피보험자와 세대를 같이하는 친족에 '
 '대한 배상책임<br>13. 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적으로 이용하는<br>중에 발생한 '
 '손해에 대한 배상책임<br>14. 가입 반려견의 소음, 냄새, 털날림으로 인하여 발생한 배상책임<br>15'),
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
