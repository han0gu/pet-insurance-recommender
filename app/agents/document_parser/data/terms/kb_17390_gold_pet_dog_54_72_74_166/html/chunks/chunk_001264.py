from langchain_core.documents import Document

chunk = Document(
    page_content=('. 의료기관의 종별은 종합병원・병원・치과병<br>원・한방병원・요양병원・정신병원・의원・치과의원・한의원 및 조산원으로 나</p><br><h1 '
 "id='86' style='font-size:14px'>누어집니다.</h1><p id='87' "
 "data-category='paragraph' style='font-size:14px'>제5조(보험금의 분담)<br>\uf000 회사는 "
 '이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약<br>을 포함합니다)이 있을 경우 각 계약에 대하여 다른 계약이 '
 '없는 것으로 하여 각<br>각 산출한'),
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
